import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS_PER_CLASS = {
    '0': 'darkgray',
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green',
    '9': 'goldenrod',
    '10': 'magenta',
    '11': 'dodgerblue',
    '12': 'darkorange',
    '13': 'olive',
    '14': 'crimson'
}


CLOUD_CLASS_INFO = {
    3: {
        "short": "CS",
        "name": "Clear sky",
        "group": "CLEAR_LOW",
        "color": "#dbe9ff",
        "order": 1,
    },
    0: {
        "short": "SC",
        "name": "Shallow clouds",
        "group": "CLEAR_LOW",
        "color": "#88a4cf",
        "order": 2,
    },
    6: {#??? not sure about that class name, it is very similar to class 5
        "short": "MC1",
        "name": "Mid-level clouds 1", #this has slightly lower CC and CTH, but higher COT than class 5
        "group": "STRATIFORM",
        "color":  "#78fba0",
        "order": 3,
    },
    5: {
        "short": "MC2",
        "name": "Mid-level clouds 2",
        "group": "STRATIFORM",
        "color": "#239b4d",
        "order": 4,
    },
    1: {
        "short": "EC",
        "name": "Early convection",
        "group": "DEVELOPING_CONVECTION",
        "color": "#fdae61",
        "order": 5,
    },
    2: {
        "short": "DC",
        "name": "Deep convection",
        "group": "DEEP_CONVECTION",
        "color": "#d73027",
        "order": 6,
    },
    4: {
        "short": "OA",
        "name": "Overcast anvils",
        "group": "ANVIL_OVERCAST",
        "color": "#8b46a1", #indigo purple
        "order": 7,
    },
}



def plot_cloud_class_legend(
    class_info: dict,
    ncols: int = 1,
    figsize=(6, 4),
    title: str = "Cloud structure classes",
    savepath: str = None,
):
    """
    Plot a standalone legend for cloud classes.

    Parameters
    ----------
    class_info : dict
        Dictionary with cloud class metadata.
        Each entry must contain:
        - 'short' : short label (e.g. CS, DCV)
        - 'name'  : long descriptive name
        - 'color' : hex color
        - 'order' : integer ordering
    ncols : int
        Number of legend columns.
    figsize : tuple
        Figure size.
    title : str
        Legend title.
    savepath : str, optional
        If provided, saves the legend to this path.
    """

    # Sort classes by physical order
    items = sorted(class_info.items(), key=lambda x: x[1]["order"])

    handles = []
    labels = []

    for cid, info in items:
        handles.append(
            Patch(
                facecolor=info["color"],
                edgecolor="black",
                linewidth=0.6
            )
        )
        labels.append(f"{info['short']} – {info['name']}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    legend = ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncols,
        frameon=False,
        title=title,
        title_fontsize=12,
        fontsize=10,
        handlelength=1.5,
        handleheight=1.2,
        labelspacing=0.8,
        columnspacing=1.5,
    )

    # Improve title appearance
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved legend to {savepath}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_cloud_class_legend(
        CLOUD_CLASS_INFO,
        ncols=1,
        figsize=(4, 3),
        title="Cloud structure classes",
        savepath="/data1/fig/dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/legend_cloud_classes.png",
    )