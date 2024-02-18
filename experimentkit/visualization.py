""" Visualization utilities"""

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns


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


def get_cmap_colors(N, cmap="brg"):
    cmap = plt.cm.get_cmap(cmap)
    return [cmap(value) for value in np.linspace(0, 1, N)]


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
