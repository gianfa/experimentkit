""" Visualization utilities"""

from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import scienceplots


def plot_n_examples(X: np.ndarray, n: int, cols: int = 2, labels: List[str] = None):
    """

    Arguments
    ---------
    X: array-like
        signals row-wise

    Examples
    --------
    >>> X = np.random.rand(10, 100)
    >>> plot_n_examples(X, 5)
    """
    X = np.array(X)
    assert X.ndim == 2, "X must be a 2-dimensional array"
    if labels:
        assert len(labels) == X.shape[0], (
            "Labels must be same length as X. "
            + "They were {len(labels)} and {X.shape[0]}"
        )
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(10, 8))
    print(f"rows: {rows}")
    fig.suptitle("Combination of Sin")
    up_limit = min(X.shape[0], n)
    for i, ax in enumerate(axs.ravel()):
        if i < up_limit:
            ax.plot(X[i, :], c="r")
            ax.set_title(f"$sig_{i}$")
            if labels:
                ax.set_sitle(f"{labels[i]}")
        else:
            ax.axis("off")
    fig.tight_layout()
    return axs


def get_cmap_colors(n: int, cmap: str = "brg") -> list:
    cmap = plt.cm.get_cmap(cmap)
    return [cmap(value) for value in np.linspace(0, 1, n)]


def plot_heatmap_discrete(
    tensor: Iterable, categories: Iterable, cmap_colors: list = None, **heatmap_kwargs
) -> plt.Axes:
    """
    Displays a heatmap using Seaborn, with distinct colors based on the
    specified categories.

    Parameters
    ----------
    tensor : torch.Tensor
        The data tensor for which to create the heatmap.
    categories : List[int]
        The list of categories for which to define
        distinct colors.
    cmap_colors: list
        The list of colors to use as a colormap. It must be as long as
        `categories`.

    Returns
    -------
    None

    Example:
    >>> tensor_data = torch.tensor([
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ...    [-2, 0.5, 1, -2, 0.5, 1, -2, 0.5, 1, -2],
    ... ])
    >>> categories = [-2, 0.5, 1]
    >>> plot_heatmap_discrete(tensor_data, categories)
    """
    if cmap_colors is None:
        cmap = sns.color_palette("rocket", n_colors=len(categories))
    else:
        cmap = sns.color_palette(cmap_colors, as_cmap=True)

    ax = sns.heatmap(tensor, cmap=cmap, vmin=0, vmax=1, fmt=".1f", **heatmap_kwargs)

    n_cats = len(categories)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    cbar_ticks = [colorbar.vmin + r / n_cats * (0.5 + i) for i in range(n_cats)]
    colorbar.set_ticks(cbar_ticks)
    colorbar.set_ticklabels(categories)
    return ax


def plot_heatmap(
    data: np.array,
    row_labels: List[str],
    col_labels: List[str],
    row_labels_fontsize: int = 11,
    col_labels_fontsize: int = 13,
    category_ranges: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrBr",
    min_max: Tuple[float] = (-1, 1),
    annot: bool = False,
    annot_kws: dict = {},
    figsize: Tuple[int] = (21, 21),
    top_labels_rotation: int = -45,
    transparent: bool = False,
    show_grid: bool = False,
    cbar_kw: dict = {},
    cbarlabel: str = "",
    im_kw: dict = {},
) -> Tuple[plt.Axes, mpl.image.AxesImage, mpl.colorbar.Colorbar]:
    """
    Create a heatmap from a numpy array and two lists of labels.

    ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    category_ranges
        If not None, a list of category ranges to create a discrete colorbar.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    nan_strategy
        A function that accepts a pandas dataframe and returns a dataframe with
        the NaN values transformed.  Optional.
    **im_kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if min_max is not None:
        vmin, vmax = min_max or (data.min().min(), data.max().max())

    # Create colorbar
    if category_ranges is not None:
        cmap = mpl.cm.get_cmap(cmap, len(category_ranges))
        norm = mpl.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = mpl.colorbar(mappable, cax=cax, **cbar_kw)
    else:
        cmap_norm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )
        cbar = ax.figure.colorbar(cmap_norm, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Create image
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **im_kw)

    # Create colorbar

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))  # , labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]))  # , labels=row_labels)
    ax.set_xticklabels(col_labels, fontsize=row_labels_fontsize)
    ax.set_yticklabels(row_labels, fontsize=col_labels_fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=top_labels_rotation,
        ha="right",
        rotation_mode="anchor",
    )

    if annot:
        # Loop over data dimensions and create text annotations.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(
                    j,
                    i,
                    f"{data[i, j]: 0.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    size=annot_kws["size"] or 12,
                )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if not transparent:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1)

    return ax, im, cbar


# ----- Styles -------
# TODO: This may be better out of this script


def get_styles(source: str = None) -> str:
    """List the availabble plotting styles"""
    splots = list(scienceplots.stylesheets.keys())
    internal = list(styles.keys())

    if source is not None:
        if source == "scienceplots":
            return splots

        elif source == "experimentkit":
            return internal

        else:
            raise ValueError(
                f"source arg must be one of {{'scienceplots', "
                + f"'experimentkit'}}; instead it was {source}"
            )

    if set(splots).intersection(set(internal)) != set():
        raise Exception("Some plot style names may be overlapping with scienceplots")
    return internal + splots


styles: Dict[str, dict] = {
    "paper-1": {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.facecolor": "white",
        "axes.titlesize": 12,
        "axes.titlepad": 6,
        "axes.titlelocation": "center",
        "axes.grid": True,
        "axes.edgecolor": "black",
        "axes.prop_cycle": plt.cycler(
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        ),
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
        # 'lines.markersize': 1,
        # 'lines.marker': 'o',
        "lines.linestyle": "-",
        "figure.facecolor": "white",
        "figure.figsize": (6, 4),
        "figure.dpi": 100,
        "legend.fontsize": 10,
    },
    "notebook": "notebook",
    "scatter": "scatter",
    "science": "science",
    "bright": "bright",
    "high-contrast": "high-contrast",
    "high-vis": "high-vis",
    "light": "light",
    "muted": "muted",
    "retro": "retro",
    "std-colors": "std-colors",
    "vibrant": "vibrant",
    "ieee": "ieee",
    "nature": "nature",
    "cjk-jp-font": "cjk-jp-font",
    "cjk-kr-font": "cjk-kr-font",
    "cjk-sc-font": "cjk-sc-font",
    "cjk-tc-font": "cjk-tc-font",
    "russian-font": "russian-font",
    "turkish-font": "turkish-font",
    "grid": "grid",
    "latex-sans": "latex-sans",
    "no-latex": "no-latex",
    "pgf": "pgf",
    "sans": "sans",
}
"""Plot Styles

A collection of ready-to-use matplotlib styles.
- `paper-1`
- *scienceplots* library styles

Example
-------
>>> import matplotlib.pyplot as plt
>>> plt.style.use(custom_style)
"""
