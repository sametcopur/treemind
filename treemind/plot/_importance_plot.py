import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from typing import Optional, Tuple, Iterable


def importance_plot(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 6.0),
    ticks_fontsize: float = 10.0,
    title_fontsize: float = 16.0,
    label_fontsize: float = 14.0,
    cmap: str = "coolwarm",
    *,
    title: Optional[str] = None,
) -> None:
    """Visualise feature-importance scores with the same visual style as
    ``interaction_plot`` and ``feature_plot``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``Result.importance(...)``. Must contain at least
        ``feature_0`` and ``importance``; ``feature_1`` triggers a
        heat-map; an optional ``class`` column creates one figure per class.
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
    required = {"feature_0", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required} columns.")

    degree_cols = [c for c in df.columns if c.startswith("feature_")]
    degree = len(degree_cols)
    if degree not in (1, 2):
        raise ValueError("importance_plot supports 1- or 2-way importances only.")

    multiclass = "class" in df.columns
    class_levels: Iterable[Optional[int]] = (
        sorted(df["class"].unique()) if multiclass else [None]
    )

    # ----------------------------- plotting ------------------------------
    for cls in class_levels:
        sub = df if cls is None else df[df["class"] == cls]

        if degree == 1:
            # -------------------------- BAR ---------------------------
            sub_sorted = sub.sort_values("importance", ascending=False)
            feat = sub_sorted["feature_0"].values
            imps = sub_sorted["importance"].values
            x = np.arange(len(feat))

            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(
                x,
                imps,
                color="#3A5894",  # same blue as other helpers
                edgecolor="white",
                width=0.7,
                zorder=3,
            )

            # Style – align with feature_plot -------------------------
            ax.set_facecolor("whitesmoke")
            ax.set_xticks(x)
            ax.set_xticklabels(feat, rotation=45, ha="right", fontsize=ticks_fontsize)
            ax.set_xlabel("Feature", fontsize=label_fontsize)
            ax.set_ylabel("Importance", fontsize=label_fontsize)

            ax.axhline(0, linestyle="--", linewidth=1, color="black", zorder=2)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            cls_txt = f" – Class {cls}" if cls is not None else ""
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
            pivot = (
                sub.pivot(index="feature_0", columns="feature_1", values="importance")
                .fillna(0.0)
            )
            # Ensure square ordering ------------------------------------------------
            feat_order = sorted(set(pivot.index).union(pivot.columns), key=str)
            pivot = pivot.reindex(index=feat_order, columns=feat_order).fillna(0.0)

            vmax = pivot.values.max()
            vmin = pivot.values.min()
            abs_max = max(abs(vmax), abs(vmin)) or 1e-8
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                pivot,
                cmap=cmap,
                norm=norm,
                center=0.0,
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Importance"},
                ax=ax,
            )

            ax.set_xlabel("Feature", fontsize=label_fontsize)
            ax.set_ylabel("Feature", fontsize=label_fontsize)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=ticks_fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=ticks_fontsize)

            cls_txt = f" – Class {cls}" if cls is not None else ""
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