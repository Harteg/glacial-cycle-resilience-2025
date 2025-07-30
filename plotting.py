import matplotlib.pyplot as plt
import numpy as np
import matplotlib

C1 = "dodgerblue"
C2 = "hotpink"

# text options
PLT_FONT_SIZE = 8 * 2


def plots(
    rows: int = 1,
    cols: int = 1,
    w: float = 15.0,
    h: float = 10.0,
    axis_labels: tuple[str | None, str | None] = (None, None),
    rotate_y_label: bool = False,
    layout: str = "constrained",
    return_ax_list: bool = False,
    h_ratios: list[float] | None = None,
    w_ratios: list[float] | None = None,
    hspace: float = 0.02,
    wspace: float = 0.02,
    hpad: float = 0.04167,
    wpad: float = 0.04167,
    **kwargs,
):
    """Plotting function.

    Args:
        rows (int): Number of rows of subplots
        cols (int): Number of columns of subplots
        w (float): Width of the figure
        h (float): Height of the figure
        axis_labels (tuple): Labels for the x and y axes
        rotate_y_label (bool): Whether to rotate the y-axis label
        layout (str): Layout of the subplots
        return_ax_list (bool): Whether to return a list of axes
        h_ratios (list): Ratios for the heights of the subplots
        w_ratios (list): Ratios for the widths of the subplots
        hspace (float): Horizontal space between subplots
        wspace (float): Vertical space between subplots
        hpad (float): Padding for the height of the subplots
        wpad (float): Padding for the width of the subplots
        **kwargs: Additional keyword arguments for plt.subplots
    """

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(w, h),
        layout=layout,
        gridspec_kw={"height_ratios": h_ratios, "width_ratios": w_ratios},
        **kwargs,
    )

    # set spacing and padding
    fig.get_layout_engine().set(hspace=hspace, wspace=wspace, h_pad=hpad, w_pad=wpad)

    # make axs a list even if only one plot
    if rows == 1 and cols == 1:
        axs = np.array([axs])

    # set ax labels if specified
    for ax in axs:
        if axis_labels[0] is not None:
            ax.set_xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            ax.set_ylabel(
                axis_labels[1],
                rotation=0 if rotate_y_label else 90,
                labelpad=10 if rotate_y_label else None,
            )

    # return ax not as list if only one plot
    if not return_ax_list and rows == 1 and cols == 1:
        axs = axs[0]

    return fig, axs


def add_text_to_ax(
    x_coord,
    y_coord,
    string,
    ax,
    fontsize=8,
    color="k",
    verticalalignment="top",
    fontfamily="monospace",
    **kwargs,
):
    """Shortcut to add text to an ax with proper font. Relative coords.

    Args:
        x_coord (float): X-coordinate of the text
        y_coord (float): Y-coordinate of the text
        string (str): Text to add
        ax (matplotlib.axes.Axes): Axes to add text to
        fontsize (float): Font size
        color (str): Color of the text
        verticalalignment (str): Vertical alignment of the text
        fontfamily (str): Font family
        **kwargs: Additional keyword arguments for plt.text
    """
    ax.text(
        x_coord,
        y_coord,
        string,
        family=fontfamily,
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment=verticalalignment,
        color=color,
        **kwargs,
    )
    return None


def add_abc_labels(axs, x=0.01, y=0.95, verticalalignment="top", add_brackets=False):
    """Add abc labels to a list of axes.

    Args:
        axs (list): List of axes
        x (float): X-coordinate of the text
        y (float): Y-coordinate of the text
        verticalalignment (str): Vertical alignment of the text
        add_brackets (bool): Whether to add brackets to the labels
    """
    abc = "abcdefghijklmnopqrstuvwxyz"
    labels = [rf"$\mathbf{{{a}}}$" for a in abc]
    for i in range(len(axs)):
        add_text_to_ax(
            x,
            y,
            "(" + labels[i] + ")" if add_brackets else labels[i],
            axs[i],
            verticalalignment=verticalalignment,
            fontsize=7,
            fontweight="bold",
        )


def get_MIS_boundaries():
    """Get MIS boundaries.
    Sourced from https://www.lorraine-lisiecki.com/LR04_MISboundaries.txt

    Returns:
        tuple: Tuple containing MIS boundaries, names, and interglacial periods
    """

    MIS_boundaries = [
        0, -14, -29, -57, -71, -130, -191, -243, -300, -337, -374, -424, -478, -533, -563, -621, -676, -712, -761, -790, -814,
    ]  # age in kyr
    MIS_names = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    ]
    MIS_interglacial = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    return MIS_boundaries, MIS_names, MIS_interglacial


def plot_MIS_boudnaries(
    ax,
    cut_at_800kyr=True,
    y_offset=-0.05,
    label_x_offset=0.0,
    label_loc="left",
    hide_label=False,
    hide_number=False,
    only_interglacial=False,
    h=0.08,
    alpha=0.15,
    font_scale=1.0
):
    """Plot MIS boundaries.

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        cut_at_800kyr (bool): Whether to cut at 800 kyr
        y_offset (float): Y-offset for the labels
        label_x_offset (float): X-offset for the label
        label_loc (str): Location of the label
        hide_label (bool): Whether to hide the label
        hide_number (bool): Whether to hide the number
        only_interglacial (bool): Whether to only plot interglacial periods
        h (float): Height of the MIS boundaries
        alpha (float): Alpha for the MIS boundaries
        font_scale (float): Font scale for the labels
    """

    MIS_boundaries, MIS_names, MIS_interglacial = get_MIS_boundaries()
    for i in range(len(MIS_boundaries) - 1):
        x1, x2 = MIS_boundaries[i], MIS_boundaries[i + 1]
        if i == 19 and cut_at_800kyr:
            x2 = -800  # set

        # plot shade if interglacial, or only_interglacial == False
        if MIS_interglacial[i] or not only_interglacial:
            ax.fill_betweenx(
                [-h + y_offset, 0 + y_offset],
                x1,
                x2,
                color=C2 if MIS_interglacial[i] else C1,
                alpha=alpha,
                edgecolor="none",
            )
            if i < 19 or not cut_at_800kyr:
                if not hide_number:
                    ax.text(
                        x1 - (x1 - x2) / 2,
                        -h / 2 + y_offset,
                        MIS_names[i],
                        ha="center",
                        va="center",
                        fontsize=PLT_FONT_SIZE * 0.4 * font_scale,
                        color="k",
                    )

    if not hide_label:
        if label_loc == "left":
            label_x = -815
        elif label_loc == "right":
            label_x = 15
        ax.text(
            label_x + label_x_offset,
            -0.05 + y_offset,
            "MIS",
            ha="center",
            va="center",
            fontsize=PLT_FONT_SIZE * 0.4 * font_scale,
            color="k",
        )

def remove_spine(ax, spine="", spines=["top", "right"]):
    """Remove spines from an ax.

    Args:
        ax (matplotlib.axes.Axes): Axes to remove spines from
        spine (str): Which spine to remove
        spines (list): List of spines to remove
    """
    if spine == "all":
        spines = ["top", "bottom", "left", "right"]
    elif spine != "":
        spines = [spine]
    for s in spines:
        ax.spines[s].set_visible(False)


def remove_spines(axs, spine="", spines=["top", "right"]):
    """Remove spines from a list of axes.

    Args:
        axs (list): List of axes
        spine (str): Which spine to remove
        spines (list): List of spines to remove
    """
    for ax in axs:
        remove_spine(ax, spine=spine, spines=spines)


# labels
def remove_tick(ax, x=True, y=True):
    """Remove ticks from an ax.

    Args:
        ax (matplotlib.axes.Axes): Axes to remove ticks from
        x (bool): Whether to remove x-ticks
        y (bool): Whether to remove y-ticks
    """
    if x:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        [label.set_visible(False) for label in ax.get_xticklabels()]
    if y:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        [label.set_visible(False) for label in ax.get_yticklabels()]


def remove_ticks(axs, x=True, y=True):
    """Remove ticks from a list of axes.

    Args:
        axs (list): List of axes
        x (bool): Whether to remove x-ticks
        y (bool): Whether to remove y-ticks
    """
    for ax in axs:
        remove_tick(ax, x=x, y=y)



# grids
def grids(
    axs, minor=True, major=True, minor_alpha=0.025, major_alpha=0.05, color="k", lw=0.5
):
    """Add grids to a list of axes.

    Args:
        axs (list): List of axes
        minor (bool): Whether to add minor grids
        major (bool): Whether to add major grids
        minor_alpha (float): Alpha for the minor grids
        major_alpha (float): Alpha for the major grids
        color (str): Color of the grids
        lw (float): Line width of the grids
    """
    for ax in axs:
        ax_add_grid(
            ax,
            minor=minor,
            major=major,
            minor_alpha=minor_alpha,
            major_alpha=major_alpha,
            color=color,
            lw=lw,
        )


def grid(
    ax, minor=True, major=True, minor_alpha=0.025, major_alpha=0.05, color="k", lw=0.5
):
    """Add grids to an ax.

    Args:
        ax (matplotlib.axes.Axes): Axes to add grids to
        minor (bool): Whether to add minor grids
        major (bool): Whether to add major grids
        minor_alpha (float): Alpha for the minor grids
        major_alpha (float): Alpha for the major grids
        color (str): Color of the grids
        lw (float): Line width of the grids
    """

    ax_add_grid(
        ax,
        minor=minor,
        major=major,
        minor_alpha=minor_alpha,
        major_alpha=major_alpha,
        color=color,
        lw=lw,
    )


def ax_add_grid(
    ax, minor=True, major=True, minor_alpha=0.025, major_alpha=0.05, color="k", lw=0.5
):
    """Add grids to an ax.

    Args:
        ax (matplotlib.axes.Axes): Axes to add grids to
        minor (bool): Whether to add minor grids
        major (bool): Whether to add major grids
        minor_alpha (float): Alpha for the minor grids
        major_alpha (float): Alpha for the major grids
        color (str): Color of the grids
        lw (float): Line width of the grids
    """

    if minor:
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(which="minor", color=color, alpha=minor_alpha, linestyle="-", lw=lw)

    if major:
        ax.grid(which="major", color=color, alpha=major_alpha, linestyle="-", lw=lw)


