import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Iterable
import re
from .. import Result
from .plot_utils import COLOR_PRIMARY


def natural_sort_key(text):
    """
    Natural sorting key function for strings containing numbers.
    Converts "Column_11" vs "Column_2" to proper numeric comparison.
    """

    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()

    return [convert(c) for c in re.split("([0-9]+)", str(text))]


def importance_plot(
    result: Result,
    combine_classes: bool = False,
    figsize: Tuple[float, float] = (10.0, 6.0),
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsize: float = 14.0,
    cmap: str = "Reds",
    *,
    title: Optional[str] = None,
) -> None:
    """Visualise feature-importance scores with the same visual style as
    ``interaction_plot`` and ``feature_plot``.

    Parameters
    ----------
    result : Result
        The result object containing the feature importance data.
    combine_classes : bool, default False
        If True, combines class-wise importances into a single DataFrame.
        Only applicable for multi-class models.
    figsize : (float, float), default ``(10, 6)``
        Figure size in inches.
    ticks_fontsize, title_fontsize, label_fontsize : float
        Font sizes for ticks, title and axis/legend labels.
    cmap : str, default ``"coolwarm"``
        Diverging colour-map centred at zero for the heat-map.
    title : str or None
        Custom figure title (string is *prepended* with class info when
        multiclass). If *None*, an automatic title is generated.
    """
    # ---------------------------- validation -----------------------------

    if not isinstance(result, Result):
        raise TypeError("result must be an instance of Result.")

    if not isinstance(combine_classes, bool):
        raise TypeError("combine_classes must be a boolean value.")

    df = result.importance(combine_classes=combine_classes)

    required = {"feature_0", "importance"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain {required} columns.")

    degree_cols = [c for c in df.columns if c.startswith("feature_")]
    degree = len(degree_cols)
    if degree not in (1, 2):
        raise ValueError("importance_plot supports 1- or 2-way importances only.")

    multiclass = "class" in df.columns
    class_levels: Iterable[Optional[int]] = (
        df["class"].unique().sort().to_list() if multiclass else [None]
    )

    # ----------------------------- plotting ------------------------------
    for cls in class_levels:
        sub = df if cls is None else df.filter(pl.col("class") == cls)

        if degree == 1:
            # -------------------------- BAR ---------------------------
            sub_sorted = sub.sort("importance", descending=True)
            feat = sub_sorted["feature_0"].to_numpy()
            imps = sub_sorted["importance"].to_numpy()
            x = np.arange(len(feat))

            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(
                x,
                imps,
                color=COLOR_PRIMARY,
                edgecolor="white",
                width=0.7,
                zorder=3,
            )

            # Style -- align with feature_plot -------------------------
            ax.set_facecolor("whitesmoke")
            ax.set_xticks(x)
            ax.set_xticklabels(feat, rotation=45, ha="right", fontsize=ticks_fontsize)
            ax.set_xlabel("Feature", fontsize=label_fontsize)
            ax.set_ylabel("Importance", fontsize=label_fontsize)

            ax.axhline(0, linestyle="--", linewidth=1, color="black", zorder=2)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            cls_txt = f" -- Class {cls}" if cls is not None else ""
            auto_title = f"Contribution of features{cls_txt}"
            ax.set_title(
                title if title is not None else auto_title,
                fontsize=title_fontsize,
                fontweight="bold",
            )

            plt.tight_layout()
            plt.show()

        else:
            # ------------------------- HEAT-MAP -----------------------
            # Build pivot table from polars: feature_0 x feature_1 -> importance
            pivot_df = sub.pivot(
                on="feature_1", index="feature_0", values="importance"
            )

            # Ensure square ordering with natural/numeric sorting --------
            feat0_vals = set(sub["feature_0"].unique().to_list())
            feat1_vals = set(sub["feature_1"].unique().to_list())
            feat_order = sorted(feat0_vals.union(feat1_vals), key=natural_sort_key)

            # Reindex: ensure all features appear as both rows and columns
            # First, get the feature_0 column and the value columns
            existing_cols = [c for c in pivot_df.columns if c != "feature_0"]

            # Add missing columns as all-null
            for f in feat_order:
                if f not in existing_cols:
                    pivot_df = pivot_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f))

            # Sort rows by feature_0 according to feat_order
            order_map = {name: i for i, name in enumerate(feat_order)}
            pivot_df = pivot_df.with_columns(
                pl.col("feature_0").replace_strict(order_map, return_dtype=pl.Int64).alias("_sort_key")
            ).sort("_sort_key").drop("_sort_key")

            # Add missing rows
            existing_feat0 = set(pivot_df["feature_0"].to_list())
            missing_rows = []
            for f in feat_order:
                if f not in existing_feat0:
                    row = {"feature_0": f}
                    for c in feat_order:
                        row[c] = None
                    missing_rows.append(row)
            if missing_rows:
                missing_df = pl.DataFrame(missing_rows, schema={"feature_0": pl.Utf8, **{f: pl.Float64 for f in feat_order}})
                pivot_df = pl.concat([pivot_df.select(["feature_0"] + feat_order), missing_df])
                # Re-sort
                pivot_df = pivot_df.with_columns(
                    pl.col("feature_0").replace_strict(order_map, return_dtype=pl.Int64).alias("_sort_key")
                ).sort("_sort_key").drop("_sort_key")

            # Extract the numpy matrix in correct column order
            pivot_matrix = pivot_df.select(feat_order).to_numpy().astype(float)

            vmax = np.nanmax(pivot_matrix)
            max_fig_size = max(figsize[0], figsize[1])
            fig, ax = plt.subplots(figsize=(max_fig_size, max_fig_size))
            sns.heatmap(
                pivot_matrix,
                cmap=cmap,
                vmin=0,  # Start from 0
                vmax=vmax,  # End at max value
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Importance"},
                ax=ax,
                xticklabels=feat_order,
                yticklabels=feat_order,
            )

            ax.set_xlabel("Feature", fontsize=label_fontsize)
            ax.set_ylabel("Feature", fontsize=label_fontsize)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=ticks_fontsize
            )
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, fontsize=ticks_fontsize
            )

            cls_txt = f" -- Class {cls}" if cls is not None else ""
            auto_title = f"Feature interaction importance{cls_txt}"
            ax.set_title(
                title if title is not None else auto_title,
                fontsize=title_fontsize,
                fontweight="bold",
                pad=12,
            )

            fig.patch.set_facecolor("whitesmoke")
            plt.tight_layout()
            plt.show()
