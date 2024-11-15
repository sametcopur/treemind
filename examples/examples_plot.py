# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go
from ipywidgets import FloatSlider, Dropdown, VBox, HBox, Layout
from IPython.display import display

def feature_plot(
    df: pd.DataFrame,
    col_max,
    col_min,
    figsize: Tuple[int, int] = (10, 6),
    show_min_max: bool = False,
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
    figsize : tuple of int, optional, default (10.0, 6.0)
        Width and height of the plot in inches.
    show_min_max : bool, optional, default False
        If True, shaded areas representing the min and max values will be displayed.
    xticks_n : int, optional, default 10
        Number of tick marks to display on the x-axis.
    yticks_n : int, optional, default 10
        Number of tick marks to display on the y-axis.
    ticks_decimal : int, optional, default 3
        Number of decimal places for tick labels
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

    # Validate parameters

    column_name = df.columns[1]

    df.loc[np.isposinf(df[column_name]), column_name] = col_max

    # Set default labels if None
    xlabel = xlabel if xlabel is not None else column_name[:-3]
    ylabel = ylabel if ylabel is not None else "Value"

    extend_ratio = 0.05 * (df[column_name].max() - df[column_name].min())

    min_row = df.iloc[df[column_name].argmin()].copy()
    min_row[column_name] = col_min

    max_row = df.iloc[df[column_name].argmax()].copy()
    max_row[column_name] += extend_ratio

    df = pd.concat([min_row.to_frame().T, df], ignore_index=True)

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=df,
        x=column_name,
        y="value",
        color="blue",
        linewidth=2,
        drawstyle="steps-pre",
    )

    if show_min_max:
        plt.fill_between(
            df[column_name],
            df["min"],
            df["max"],
            color="gray",
            alpha=0.3,
            label="Min-Max Range",
            step="post",
        )

    # Set the plot title
    if title is None:
        plt.title(
            f"Contribution of {xlabel}", fontsize=title_fontsize, fontweight="bold"
        )
    else:
        plt.title(title, fontsize=title_fontsize, fontweight="bold")

    plt.gca().set_facecolor("whitesmoke")

    plt.xlabel(xlabel, fontsize=label_fontsizes)
    plt.ylabel(ylabel, fontsize=label_fontsizes)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if show_min_max:
        plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Functions for plotting real data:
# - _set_common_style: Applies a consistent visual style to all plots to match the style used by TreeMind.
# - plot_line_chart: Creates a line chart to visualize trends in data over a continuous range.
# - plot_scatter_chart: Generates a scatter plot to explore relationships between two variables, with a third variable indicated by color.


def _set_common_style(ax, title):
    ax.set_facecolor("whitesmoke")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_line_chart(x, y, title, x_label, y_label, figsize: Tuple[int, int] = (10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    # lineplot yerine scatterplot kullanıyoruz
    sns.scatterplot(x=x, y=y, ax=ax, color="blue")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label) 
    _set_common_style(ax, title)
    plt.tight_layout()
    plt.show()


def plot_scatter_chart(
    x, y, z, title, x_label, y_label, cbar_label, figsize: Tuple[int, int] = (10, 6)
):
    fig, ax = plt.subplots(figsize=figsize)

    max_val = z.max()
    min_val = z.min()

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

    scatter = ax.scatter(x, y, c=z, cmap=colormap, norm=norm, edgecolor="k")

    cbar = plt.colorbar(scatter, ax=ax)

    cbar.ax.set_yscale("linear")
    cbar.ax.set_ylim(min_val, max_val)
    cbar.set_label(cbar_label)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    _set_common_style(ax, title)

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
    #cbar.ax.set_yscale("linear")
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

def scatter_3d_plot_test(
    x,
    y,
    z,
    value,
    width=1000,
    height=800,
    opacity=0.8,
    point_size=5
):
    """
    Enhanced 3D scatter plot with smooth color transitions.
    Red indicates positive values, blue indicates negative values, yellow for zero.

    Parameters:
    - x, y, z (array-like): Coordinates for the 3D plot.
    - value (array-like): Values for color intensity, typically associated with (x, y, z) points.
    - width (int): Width of the plot. Default is 1000
    - height (int): Height of the plot. Default is 800
    - opacity (float): Opacity of the scatter points. Default is 0.8
    - point_size (int): Size of scatter points. Default is 5
    """

    # Updated color scale with blue-yellow-red transition
    colorscale = [
        [0.0, 'rgb(0,0,255)'],
        [0.5, 'rgb(255,255,0)'],
        [1.0, 'rgb(255,0,0)']
    ]
    
    # Get value range for consistent color scaling
    value_min = min(value)
    value_max = max(value)
    abs_max = max(abs(value_min), abs(value_max))
    cmin, cmax = -abs_max, abs_max

    # Create figure
    fig = go.FigureWidget()

    # Add scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=point_size,
                color=value,
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
                "<b>X</b>: %{x:.2f}<br>"
                "<b>Y</b>: %{y:.2f}<br>"
                "<b>Z</b>: %{z:.2f}<br>"
                "<b>Value</b>: %{marker.color:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
                titlefont=dict(size=15),
                range=[min(x), max(x)],
            ),
            yaxis=dict(
                title="Y",
                titlefont=dict(size=15),
                range=[min(y), max(y)],
            ),
            zaxis=dict(
                title="Z",
                titlefont=dict(size=15),
                range=[min(z), max(z)],
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
        value=value_min,
        min=value_min,
        max=value_max,
        step=(value_max - value_min) / 100,
        description="Min Intensity",
        style={'description_width': '100px'},
        layout=Layout(width='400px'),
    )
    
    max_slider = FloatSlider(
        value=value_max,
        min=value_min,
        max=value_max,
        step=(value_max - value_min) / 100,
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

        mask = [(val >= min_intensity) & (val <= max_intensity) for val in value]
        filtered_x = [x[i] for i, m in enumerate(mask) if m]
        filtered_y = [y[i] for i, m in enumerate(mask) if m]
        filtered_z = [z[i] for i, m in enumerate(mask) if m]
        filtered_value = [value[i] for i, m in enumerate(mask) if m]

        with fig.batch_update():
            fig.data[0].x = filtered_x
            fig.data[0].y = filtered_y
            fig.data[0].z = filtered_z
            fig.data[0].marker.color = filtered_value
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
